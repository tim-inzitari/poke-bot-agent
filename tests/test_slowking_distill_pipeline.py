"""Unit tests for the Slowking research distillation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poke_bot.slowking_distill.authority import RESEARCH_ONLY, TRAINING_AUTHORITY
from poke_bot.slowking_distill.bc_stage import StageAConfig, run_stage_a_bc
from poke_bot.slowking_distill.critical_search import (
    is_critical_decision,
    mock_top_k_search,
    run_stage_c_search,
)
from poke_bot.slowking_distill.day_split import build_day_split, default_val_dates_for_window
from poke_bot.slowking_distill.distill_search import run_stage_e_distill
from poke_bot.slowking_distill.eval_gate import evaluate_paired_games
from poke_bot.slowking_distill.heuristic_features import attach_heuristic_features
from poke_bot.slowking_distill.iql import expectile_loss, run_stage_b_iql
from poke_bot.slowking_distill.multi_day import aggregate_daily_receipts
from poke_bot.slowking_distill.pipeline import PipelineConfig, run_pipeline
from poke_bot.slowking_distill.bc_stage import OptionConditionedClone
from poke_bot.slowking_reverse_engineered_policy import (
    ACADEMY_AT_NIGHT,
    CTX_SETUP_ACTIVE,
    CTX_TOP_DECK,
    KYUREM,
    MEGA_KANGASKHAN_EX,
    OPT_PLAY,
    SLOWKING,
    SLOWPOKE,
    SMOOCHUM,
)


ROOT = Path(__file__).resolve().parents[1]


def _deck() -> list[int]:
    final = json.loads(
        (ROOT / "state" / "slowking_top_replay_distillation_2026-08-04.json").read_text(
            encoding="utf-8"
        )
    )
    counts = {
        row["card_id"]: row["count"] for row in final["identity"]["decks"][0]["cards"]
    }
    out: list[int] = []
    for card_id, count in sorted(counts.items()):
        out.extend([card_id] * count)
    return out


DECK = _deck()


def _opening_row(*, game_id: str, date: str, prefer: int = MEGA_KANGASKHAN_EX) -> dict:
    hand = [prefer, SMOOCHUM, SLOWPOKE, KYUREM]
    options = [{"type": OPT_PLAY, "index": i} for i in range(4)]
    obs = {
        "current": {
            "yourIndex": 0,
            "firstPlayer": 0,
            "turn": 0,
            "players": [
                {
                    "active": [],
                    "bench": [],
                    "hand": [{"id": c} for c in hand],
                    "discard": [],
                },
                {"active": [], "bench": [], "hand": [], "discard": []},
            ],
        },
        "select": {"context": CTX_SETUP_ACTIVE, "option": options, "minCount": 1, "maxCount": 1},
    }
    legal = [[0], [1], [2], [3]]
    selected = 0
    return {
        "game_id": game_id,
        "source_date": date,
        "episode_id": game_id.split(":")[1] if ":" in game_id else game_id,
        "seat": 0,
        "env_step": 0,
        "deck": list(DECK),
        "result": "win",
        "reward": 1,
        "value_target": 1.0,
        "turn_order": "first",
        "observation": obs,
        "action": legal[selected],
        "legal_action_combos": legal,
        "selected_index": selected,
        "legal_action_count": 4,
    }


def _academy_row(*, game_id: str, date: str) -> dict:
    options = [{"type": OPT_PLAY, "index": i} for i in range(2)]
    obs = {
        "current": {
            "yourIndex": 0,
            "firstPlayer": 0,
            "turn": 2,
            "players": [
                {
                    "active": [{"id": SLOWKING}],
                    "bench": [{"id": SLOWPOKE}],
                    "hand": [{"id": KYUREM}, {"id": SLOWPOKE}],
                    "discard": [],
                },
                {"active": [], "bench": [], "hand": [], "discard": []},
            ],
        },
        "select": {
            "context": CTX_TOP_DECK,
            "effect": {"id": ACADEMY_AT_NIGHT},
            "option": options,
            "minCount": 1,
            "maxCount": 1,
        },
    }
    legal = [[0], [1]]
    return {
        "game_id": game_id,
        "source_date": date,
        "episode_id": "ep",
        "seat": 0,
        "env_step": 3,
        "deck": list(DECK),
        "result": "loss",
        "reward": -1,
        "value_target": -1.0,
        "turn_order": "second",
        "observation": obs,
        "action": [0],
        "legal_action_combos": legal,
        "selected_index": 0,
        "legal_action_count": 2,
    }


@pytest.mark.unit
def test_authority_defaults() -> None:
    assert RESEARCH_ONLY is True
    assert TRAINING_AUTHORITY is False


@pytest.mark.unit
def test_day_split_holds_out_entire_days() -> None:
    games = [
        {"game_id": "2026-08-02:a:0", "source_date": "2026-08-02"},
        {"game_id": "2026-08-03:b:0", "source_date": "2026-08-03"},
        {"game_id": "2026-08-04:c:0", "source_date": "2026-08-04"},
    ]
    split = build_day_split(games, val_dates=["2026-08-04"])
    assert "2026-08-04" in split.val_dates
    assert "2026-08-04:c:0" in split.val_game_ids
    assert "2026-08-04:c:0" not in split.train_game_ids
    assert default_val_dates_for_window(["2026-08-02", "2026-08-04"]) == ["2026-08-04"]


@pytest.mark.unit
def test_heuristic_attach_and_zero_ablation() -> None:
    row = _opening_row(game_id="2026-08-02:g:0", date="2026-08-02")
    attached = attach_heuristic_features(row)
    assert attached["heuristic_abstained"] is False
    assert attached["heuristic"]["stage_class"] == "opening_active"
    zeroed = attach_heuristic_features(row, zero_channel=True)
    assert zeroed["heuristic_channel_zeroed"] is True
    assert zeroed["heuristic"]["scores"] == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.unit
def test_stage_a_and_b_and_c_and_e(tmp_path: Path) -> None:
    rows = [
        _opening_row(game_id=f"2026-08-02:g{i}:0", date="2026-08-02")
        for i in range(8)
    ] + [
        _academy_row(game_id=f"2026-08-03:a{i}:0", date="2026-08-03")
        for i in range(4)
    ]
    # Include losses explicitly.
    assert any(r["result"] == "loss" for r in rows)

    a = run_stage_a_bc(rows, out_dir=tmp_path / "a", config=StageAConfig(epochs=1, batch_size=4))
    assert a.metrics["n_rows"] == len(rows)
    assert Path(a.checkpoint_path).is_file()

    actor = OptionConditionedClone(64)
    actor.load_state_dict(
        torch.load(a.checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    )
    b = run_stage_b_iql(rows, actor=actor, out_dir=tmp_path / "b")
    assert "rank_correlation" in b.metrics

    critical, reason = is_critical_decision(attach_heuristic_features(rows[0]))
    assert critical
    assert "heuristic_stage" in reason
    c = run_stage_c_search(rows, out_dir=tmp_path / "c")
    assert c["n_receipts"] >= 1
    receipts = [
        json.loads(line)
        for line in Path(c["receipts_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    e = run_stage_e_distill(rows, receipts, actor=actor, out_dir=tmp_path / "e", epochs=1)
    assert e.metrics["n_paired"] >= 1


@pytest.mark.unit
def test_expectile_loss_asymmetric() -> None:
    diff = torch.tensor([1.0, -1.0])
    loss = expectile_loss(diff, 0.9)
    assert float(loss.item()) > 0


@pytest.mark.unit
def test_eval_gate_rejects_agreement_only_and_low_n() -> None:
    result = evaluate_paired_games([], action_agreement=0.99)
    assert result.passed is False
    assert any("insufficient_games" in r for r in result.reasons)

    games = [
        {"result": "win", "training_ineligible": True}
        for _ in range(70)
    ]
    ok = evaluate_paired_games(games)
    assert ok.passed is True


@pytest.mark.unit
def test_pipeline_end_to_end(tmp_path: Path) -> None:
    rows = []
    for i in range(6):
        rows.append(_opening_row(game_id=f"2026-08-02:t{i}:0", date="2026-08-02"))
    for i in range(4):
        rows.append(_academy_row(game_id=f"2026-08-04:v{i}:0", date="2026-08-04"))
    jsonl = tmp_path / "decisions.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = run_pipeline(
        jsonl,
        config=PipelineConfig(
            out_dir=tmp_path / "run",
            val_dates=["2026-08-04"],
            stage_a=StageAConfig(epochs=1, batch_size=4),
            max_critical_search=8,
            eval_games=[],  # fail-closed on paired gate
        ),
    )
    assert summary["research_only"] is True
    assert summary["promoted"] is False
    assert summary["eval_gate"]["passed"] is False
    assert Path(summary["day_split_path"]).is_file()


@pytest.mark.unit
def test_aggregate_daily_receipt(tmp_path: Path) -> None:
    src = ROOT / "state" / "slowking_top_replay_distillation_2026-08-04.json"
    # Minimal second daily clone with different date for aggregation shape.
    daily = json.loads(src.read_text(encoding="utf-8"))
    daily["source"]["date"] = "2026-08-03"
    daily["games"] = daily["games"][:2]
    daily["outcomes"]["games"] = 2
    p1 = tmp_path / "d1.json"
    p2 = tmp_path / "d2.json"
    p1.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    p2.write_text(json.dumps(daily), encoding="utf-8")
    agg = aggregate_daily_receipts([p1, p2], window_start="2026-08-03", window_end="2026-08-04")
    assert agg["identity"]["slowking_seats"] == len(daily["games"]) + len(
        json.loads(src.read_text())["games"]
    )
    assert agg["training_authority"] is False


@pytest.mark.unit
def test_mock_search_receipt_digest() -> None:
    row = attach_heuristic_features(_academy_row(game_id="2026-08-04:x:0", date="2026-08-04"))
    receipt = mock_top_k_search(row)
    assert receipt.payload["receipt_sha256"].startswith("sha256:")
    assert receipt.payload["chosen_action"] == [0]
