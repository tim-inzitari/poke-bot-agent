from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.select_next_specialist import (
    FALLBACK_SOURCE_MODE,
    select,
    validate_corpus_source_contract,
)


def _state(path: Path) -> Path:
    ids = ["dudunsparce", "lucario"] + [f"target-{i}" for i in range(17)]
    specialists = [
        {"id": "alakazam", "status": "passed_frozen"},
        {"id": "hops-trevenant", "status": "passed_frozen"},
        {"id": "starmie", "status": "rl_training"},
        *[{"id": value, "status": "unstarted"} for value in ids],
    ]
    roster_ids = [str(row["id"]) for row in specialists]
    (path.parent / "matchup_adapter_roster.json").write_text(
        json.dumps(
            {
                "required_specialist_count": len(roster_ids),
                "physical_checkpoint_rows": len(roster_ids),
                "expert_ids": roster_ids,
            }
        ),
        encoding="utf-8",
    )
    path.write_text(
        yaml.safe_dump(
            {
                "current": {"program_progress": {"remaining_after_active": 19}},
                "target_registry": {"required_target_count": 22},
                "training_priority": {
                    "ordered_unfinished_ids_after_active": ids
                },
                "specialists": specialists,
            }
        ),
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _latest20_manifest(specialist_id: str, *, matching_games: int = 1) -> dict:
    dates = [f"2026-07-{day:02d}" for day in range(1, 21)]
    return {
        "selection": {"value": specialist_id},
        "source_window": {
            "unit": "calendar_day",
            "selection": "latest_available_fully_validated_daily_sources",
            "days": 20,
            "dates": dates,
            "filter_applied_after_window_selection": True,
            "filter_archetype": specialist_id,
        },
        "source_days": [
            {
                "date": value,
                "source_feature_validated": True,
                "source_feature_sha256": "sha256:" + "a" * 64,
                "source_archive_validated": True,
                "source_archive_sha256": "sha256:" + "b" * 64,
                "matching_games": matching_games if index == 0 else 0,
                "matching_decisions": 8 if index == 0 and matching_games else 0,
            }
            for index, value in enumerate(dates)
        ],
    }


def _corpus(root: Path, specialist_id: str, decisions: int) -> None:
    directory = root / specialist_id
    directory.mkdir(parents=True)
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(_latest20_manifest(specialist_id)),
        encoding="utf-8",
    )
    (directory / "PROTECTED_EXPERT_CORPUS.json").write_text(
        json.dumps(
            {
                "schema": "poke_bot.pinned_expert_corpus/v1",
                "protected": True,
                "manifest": manifest.name,
                "manifest_sha256": _sha256(manifest),
                "totals": {"decisions_kept": decisions},
            }
        ),
        encoding="utf-8",
    )


def _guide_corpus(root: Path, specialist_id: str, decisions: int) -> Path:
    directory = root / specialist_id
    directory.mkdir(parents=True)
    dates = [f"2026-07-{day:02d}" for day in range(1, 21)]
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "selection": {"value": specialist_id},
                "totals": {"decisions_kept": decisions},
            }
        ),
        encoding="utf-8",
    )
    daily = []
    per_day = decisions // len(dates)
    for source_date in dates:
        feature = directory / f"{specialist_id}-{source_date}.features"
        feature.write_bytes(f"feature-{source_date}".encode())
        receipt = directory / (
            f"{specialist_id}-{source_date}.features.receipt.json"
        )
        receipt.write_text(
            json.dumps(
                {
                    "format": "pokebot-authoritative-visual-day-receipt",
                    "source_date": source_date,
                    "selection": {
                        "acting_seat_archetype": specialist_id,
                        "current_deck_guide": specialist_id,
                    },
                    "source_archive": {"sha256": "sha256:" + "b" * 64},
                    "output": {"sha256": _sha256(feature)},
                    "stats": {
                        "records_kept": 1,
                        "decisions_kept": per_day,
                    },
                }
            ),
            encoding="utf-8",
        )
        daily.append(
            {
                "date": source_date,
                "records": 1,
                "decisions": per_day,
                "sha256": _sha256(feature),
                "receipt_sha256": _sha256(receipt),
            }
        )
    ready = directory / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    ready.write_text(
        json.dumps(
            {
                "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
                "status": "ready",
                "specialist_id": specialist_id,
                "days": 20,
                "dates": dates,
                "manifest_sha256": _sha256(manifest),
                "daily_shards": daily,
            }
        ),
        encoding="utf-8",
    )
    pointer = directory / "PROTECTED_EXPERT_CORPUS.json"
    pointer.write_text(
        json.dumps(
            {
                "schema": "poke_bot.pinned_expert_corpus/v1",
                "protected": True,
                "manifest": manifest.name,
                "manifest_sha256": _sha256(manifest),
                "totals": {"decisions_kept": decisions},
            }
        ),
        encoding="utf-8",
    )
    return pointer


def _fallback_corpus(
    root: Path,
    specialist_id: str,
    decisions: int,
    *,
    latest20_matching_games: int = 0,
) -> Path:
    directory = root / specialist_id
    directory.mkdir(parents=True)
    shard = directory / "historical.features"
    shard.write_bytes(b"validated historical feature shard")
    historical = directory / "historical-manifest.json"
    historical.write_text(
        json.dumps(
            {
                "selection": {"value": specialist_id},
                "quality_gates": {"passed": True, "checksummed": True},
                "shards": [
                    {
                        "path": shard.name,
                        "bytes": shard.stat().st_size,
                        "sha256": _sha256(shard),
                    }
                ],
                "totals": {"decisions_kept": decisions},
            }
        ),
        encoding="utf-8",
    )
    evidence = directory / "latest20-zero-match-evidence.json"
    evidence.write_text(
        json.dumps(
            _latest20_manifest(
                specialist_id,
                matching_games=latest20_matching_games,
            )
        ),
        encoding="utf-8",
    )
    pointer = directory / "PROTECTED_EXPERT_CORPUS.json"
    pointer.write_text(
        json.dumps(
            {
                "schema": "poke_bot.pinned_expert_corpus/v1",
                "protected": True,
                "manifest": historical.name,
                "manifest_sha256": _sha256(historical),
                "totals": {"decisions_kept": decisions},
                "source_policy": {
                    "mode": FALLBACK_SOURCE_MODE,
                    "reason": "latest20_matching_games_exactly_zero",
                    "fallback_is_latest20": False,
                    "latest20_zero_match_manifest": evidence.name,
                    "latest20_zero_match_manifest_sha256": _sha256(evidence),
                },
            }
        ),
        encoding="utf-8",
    )
    return pointer


def test_selection_defers_required_but_insufficient_target(tmp_path: Path) -> None:
    state = _state(tmp_path / "state.yaml")
    corpora = tmp_path / "corpora"
    _corpus(corpora, "dudunsparce", 2_448)
    _corpus(corpora, "lucario", 173_490)

    result = select(
        state_path=state,
        corpus_root=corpora,
        minimum_decisions=20_000,
    )

    assert result["required_specialists_total"] == 22
    assert result["remaining_after_starmie"] == 19
    assert result["selected"]["specialist_id"] == "lucario"
    assert result["deferred_higher_priority"] == [
        {
            "specialist_id": "dudunsparce",
            "priority_rank": 0,
            "reason": "protected_expert_corpus_below_contract",
            "decisions": 2448,
            "minimum_decisions": 20000,
            "pointer": str(
                corpora.resolve()
                / "dudunsparce"
                / "PROTECTED_EXPERT_CORPUS.json"
            ),
        }
    ]


def test_staged_successor_is_stable_when_higher_priority_becomes_ready(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "state.yaml")
    payload = yaml.safe_load(state.read_text(encoding="utf-8"))
    payload["current"]["staged_successor_specialist"] = "lucario"
    state.write_text(yaml.safe_dump(payload), encoding="utf-8")
    corpora = tmp_path / "corpora"
    _corpus(corpora, "dudunsparce", 24_000)
    _corpus(corpora, "lucario", 24_000)

    result = select(
        state_path=state,
        corpus_root=corpora,
        minimum_decisions=20_000,
    )

    assert result["selected"]["specialist_id"] == "lucario"
    assert result["staged_successor_specialist"] == "lucario"
    assert result["deferred_higher_priority"] == []


def test_staged_successor_fails_closed_when_its_corpus_is_not_ready(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "state.yaml")
    payload = yaml.safe_load(state.read_text(encoding="utf-8"))
    payload["current"]["staged_successor_specialist"] = "lucario"
    state.write_text(yaml.safe_dump(payload), encoding="utf-8")
    corpora = tmp_path / "corpora"
    _corpus(corpora, "dudunsparce", 24_000)

    with pytest.raises(
        RuntimeError,
        match=(
            "staged successor lucario is not executable: "
            "protected_expert_corpus_missing"
        ),
    ):
        select(
            state_path=state,
            corpus_root=corpora,
            minimum_decisions=20_000,
        )


def test_stale_staged_successor_equal_to_active_is_ignored(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "state.yaml")
    payload = yaml.safe_load(state.read_text(encoding="utf-8"))
    payload["current"]["staged_successor_specialist"] = "dudunsparce"
    state.write_text(yaml.safe_dump(payload), encoding="utf-8")
    corpora = tmp_path / "corpora"
    _corpus(corpora, "lucario", 24_000)

    result = select(
        state_path=state,
        corpus_root=corpora,
        minimum_decisions=20_000,
        active_id="dudunsparce",
    )

    assert result["selected"]["specialist_id"] == "lucario"
    assert result["staged_successor_specialist"] is None


def test_completed_target_is_skipped_without_becoming_deferred(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "state.yaml")
    payload = yaml.safe_load(state.read_text(encoding="utf-8"))
    next(
        row for row in payload["specialists"] if row["id"] == "dudunsparce"
    )["status"] = "passed_frozen"
    state.write_text(yaml.safe_dump(payload), encoding="utf-8")
    corpora = tmp_path / "corpora"
    _corpus(corpora, "lucario", 20_000)

    result = select(
        state_path=state,
        corpus_root=corpora,
        minimum_decisions=20_000,
    )

    assert result["selected"]["specialist_id"] == "lucario"
    assert result["deferred_higher_priority"] == []


def test_runtime_frozen_registry_override_skips_stale_yaml_status(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "state.yaml")
    corpora = tmp_path / "corpora"
    _corpus(corpora, "dudunsparce", 25_000)
    _corpus(corpora, "lucario", 25_000)

    result = select(
        state_path=state,
        corpus_root=corpora,
        minimum_decisions=20_000,
        completed_ids={"dudunsparce"},
        active_id="starmie",
    )

    assert result["selected"]["specialist_id"] == "lucario"
    assert result["completed_specialist_ids"] == ["dudunsparce"]


def test_registered_completion_normalizes_stale_priority_projection(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "state.yaml")
    payload = yaml.safe_load(state.read_text(encoding="utf-8"))
    next(
        row for row in payload["specialists"] if row["id"] == "dudunsparce"
    )["status"] = "passed_frozen"
    # Registration and the remaining count are authoritative at this boundary,
    # while the mutable ordered projection can lag until its next reconciliation.
    payload["current"]["program_progress"]["remaining_after_active"] = 18
    state.write_text(yaml.safe_dump(payload), encoding="utf-8")
    corpora = tmp_path / "corpora"
    _corpus(corpora, "lucario", 25_000)

    result = select(
        state_path=state,
        corpus_root=corpora,
        minimum_decisions=20_000,
        completed_ids={"dudunsparce"},
        active_id="starmie",
    )

    assert result["remaining_unfinished"] == 18
    assert result["selected"]["specialist_id"] == "lucario"
    assert result["completed_specialist_ids"] == ["dudunsparce"]


def test_selection_accepts_live_post_activation_unfinished_count(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "state.yaml")
    payload = yaml.safe_load(state.read_text(encoding="utf-8"))
    next(
        row for row in payload["specialists"] if row["id"] == "lucario"
    )["status"] = "rl_training"
    payload["training_priority"]["ordered_unfinished_ids_after_active"].remove(
        "lucario"
    )
    payload["current"]["program_progress"]["remaining_after_active"] = 18
    state.write_text(yaml.safe_dump(payload), encoding="utf-8")
    corpora = tmp_path / "corpora"
    _corpus(corpora, "dudunsparce", 25_000)

    result = select(
        state_path=state,
        corpus_root=corpora,
        minimum_decisions=20_000,
        completed_ids={"alakazam", "hops-trevenant", "starmie"},
        active_id="lucario",
    )

    assert result["remaining_unfinished"] == 18
    assert result["selected"]["specialist_id"] == "dudunsparce"
    assert result["active_specialist_id"] == "lucario"


def test_selection_accepts_canonical_boundary_remaining_unfinished_name(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "state.yaml")
    payload = yaml.safe_load(state.read_text(encoding="utf-8"))
    progress = payload["current"]["program_progress"]
    progress["remaining_unfinished"] = progress.pop("remaining_after_active")
    state.write_text(yaml.safe_dump(payload), encoding="utf-8")
    corpora = tmp_path / "corpora"
    _corpus(corpora, "dudunsparce", 25_000)

    result = select(
        state_path=state,
        corpus_root=corpora,
        minimum_decisions=20_000,
    )

    assert result["selected"]["specialist_id"] == "dudunsparce"


def test_unvalidated_router_route_is_deferred(tmp_path: Path) -> None:
    state = _state(tmp_path / "state.yaml")
    corpora = tmp_path / "corpora"
    _corpus(corpora, "dudunsparce", 25_000)
    _corpus(corpora, "lucario", 25_000)

    result = select(
        state_path=state,
        corpus_root=corpora,
        minimum_decisions=20_000,
        routable_ids={"lucario"},
    )

    assert result["selected"]["specialist_id"] == "lucario"
    assert result["deferred_higher_priority"] == [
        {
            "specialist_id": "dudunsparce",
            "priority_rank": 0,
            "reason": "validated_causal_runtime_route_missing",
        }
    ]


def test_specialist_minimum_override_selects_strict_priority(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "state.yaml")
    corpora = tmp_path / "corpora"
    _corpus(corpora, "dudunsparce", 2_448)
    _corpus(corpora, "lucario", 25_000)

    result = select(
        state_path=state,
        corpus_root=corpora,
        minimum_decisions=20_000,
        minimum_decisions_by_specialist={"dudunsparce": 2_000},
        strict_priority_prefix=["dudunsparce"],
    )

    assert result["selected"]["specialist_id"] == "dudunsparce"
    assert result["selected"]["minimum_decisions"] == 2_000


def test_strict_priority_cannot_silently_fall_through(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "state.yaml")
    corpora = tmp_path / "corpora"
    _corpus(corpora, "dudunsparce", 1_999)
    _corpus(corpora, "lucario", 25_000)

    try:
        select(
            state_path=state,
            corpus_root=corpora,
            minimum_decisions=20_000,
            minimum_decisions_by_specialist={"dudunsparce": 2_000},
            strict_priority_prefix=["dudunsparce"],
        )
    except RuntimeError as error:
        assert "strict priority specialist dudunsparce" in str(error)
    else:
        raise AssertionError("strict priority target was silently skipped")


def test_latest20_requires_all_consecutive_days_and_post_window_filter(
    tmp_path: Path,
) -> None:
    corpora = tmp_path / "corpora"
    _corpus(corpora, "dudunsparce", 25_000)
    pointer = corpora / "dudunsparce" / "PROTECTED_EXPERT_CORPUS.json"
    validated = validate_corpus_source_contract(
        pointer,
        specialist_id="dudunsparce",
    )
    assert validated["mode"] == "latest20_primary"
    assert validated["latest20"]["days"] == 20
    assert validated["latest20"]["all_dates_represented"] is True
    assert validated["latest20"]["filter_applied_after_window_selection"] is True

    manifest = corpora / "dudunsparce" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_days"].pop()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    pointer_payload["manifest_sha256"] = _sha256(manifest)
    pointer.write_text(json.dumps(pointer_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact latest-20"):
        validate_corpus_source_contract(pointer, specialist_id="dudunsparce")


def test_guide_corpus_proves_latest20_through_all_daily_receipts(
    tmp_path: Path,
) -> None:
    pointer = _guide_corpus(
        tmp_path / "corpora",
        "marnie-s-grimmsnarl-ex",
        40_000,
    )
    validated = validate_corpus_source_contract(
        pointer,
        specialist_id="marnie-s-grimmsnarl-ex",
    )
    latest20 = validated["latest20"]
    assert latest20["days"] == 20
    assert latest20["matching_games"] == 20
    assert latest20["matching_decisions"] == 40_000
    assert latest20["evidence_source"] == (
        "current_deck_guide_ready_receipt_and_daily_receipts"
    )

    receipt = pointer.parent / (
        "marnie-s-grimmsnarl-ex-2026-07-01.features.receipt.json"
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["source_archive"]["sha256"] = "not-a-checksum"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    ready = pointer.parent / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    ready_payload = json.loads(ready.read_text(encoding="utf-8"))
    ready_payload["daily_shards"][0]["receipt_sha256"] = _sha256(receipt)
    ready.write_text(json.dumps(ready_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="daily receipt is invalid"):
        validate_corpus_source_contract(
            pointer,
            specialist_id="marnie-s-grimmsnarl-ex",
        )


def test_historical_fallback_requires_exactly_zero_latest20_matches(
    tmp_path: Path,
) -> None:
    zero_pointer = _fallback_corpus(
        tmp_path / "zero",
        "dudunsparce",
        25_000,
    )
    validated = validate_corpus_source_contract(
        zero_pointer,
        specialist_id="dudunsparce",
    )
    assert validated["mode"] == FALLBACK_SOURCE_MODE
    assert validated["historical_fallback"] is True
    assert validated["masquerades_as_latest20"] is False
    assert validated["latest20"]["matching_games"] == 0

    nonzero_pointer = _fallback_corpus(
        tmp_path / "nonzero",
        "dudunsparce",
        25_000,
        latest20_matching_games=1,
    )
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_corpus_source_contract(
            nonzero_pointer,
            specialist_id="dudunsparce",
        )


def test_selection_receipt_exposes_fallback_instead_of_latest20_masquerade(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path / "state.yaml")
    corpora = tmp_path / "corpora"
    _fallback_corpus(corpora, "dudunsparce", 25_000)

    result = select(
        state_path=state,
        corpus_root=corpora,
        minimum_decisions=20_000,
    )

    source = result["selected"]["source_contract"]
    assert source["mode"] == FALLBACK_SOURCE_MODE
    assert source["historical_fallback"] is True
    assert source["masquerades_as_latest20"] is False
