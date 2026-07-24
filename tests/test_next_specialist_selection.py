from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.select_next_specialist import select


def _state(path: Path) -> Path:
    ids = ["dudunsparce", "lucario"] + [f"target-{i}" for i in range(17)]
    specialists = [
        {"id": "alakazam", "status": "passed_frozen"},
        {"id": "hops-trevenant", "status": "passed_frozen"},
        {"id": "starmie", "status": "rl_training"},
        *[{"id": value, "status": "unstarted"} for value in ids],
    ]
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


def _corpus(root: Path, specialist_id: str, decisions: int) -> None:
    directory = root / specialist_id
    directory.mkdir(parents=True)
    (directory / "PROTECTED_EXPERT_CORPUS.json").write_text(
        json.dumps(
            {
                "schema": "poke_bot.pinned_expert_corpus/v1",
                "protected": True,
                "totals": {"decisions_kept": decisions},
            }
        ),
        encoding="utf-8",
    )


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
