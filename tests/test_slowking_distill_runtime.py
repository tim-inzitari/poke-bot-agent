"""Tests for Slowking distill runtime, self-play, promotion, and PolicyAgent bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.slowking_distill.bc_stage import StageAConfig, run_stage_a_bc
from poke_bot.slowking_distill.belief_search_backend import (
    BeliefSearchBundle,
    resolve_stage_c_search_fn,
)
from poke_bot.slowking_distill.config import (
    SlowkingDistillMode,
    SlowkingDistillRuntimeConfig,
    load_runtime_config_from_env,
)
from poke_bot.slowking_distill.heuristic_features import attach_heuristic_features
from poke_bot.slowking_distill.policy_bridge import SlowkingDistillAgentBridge
from poke_bot.slowking_distill.promotion import PromotionRequest, evaluate_promotion
from poke_bot.slowking_distill.runtime import SlowkingDistillRuntime
from poke_bot.slowking_distill.self_play import run_population_self_play
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


def _opening_row() -> dict:
    hand = [MEGA_KANGASKHAN_EX, SMOOCHUM, SLOWPOKE, KYUREM]
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
        "select": {
            "context": CTX_SETUP_ACTIVE,
            "option": options,
            "minCount": 1,
            "maxCount": 1,
        },
    }
    legal = [[0], [1], [2], [3]]
    return {
        "game_id": "2026-08-02:rt:0",
        "source_date": "2026-08-02",
        "episode_id": "rt",
        "seat": 0,
        "env_step": 0,
        "deck": list(DECK),
        "result": "win",
        "reward": 1,
        "value_target": 1.0,
        "turn_order": "first",
        "observation": obs,
        "action": [0],
        "legal_action_combos": legal,
        "selected_index": 0,
        "legal_action_count": 4,
    }


def _academy_row() -> dict:
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
                    "hand": [{"id": KYUREM}],
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
    return {
        "game_id": "2026-08-02:ac:0",
        "source_date": "2026-08-02",
        "episode_id": "ac",
        "seat": 0,
        "env_step": 3,
        "deck": list(DECK),
        "result": "win",
        "reward": 1,
        "observation": obs,
        "action": [0],
        "legal_action_combos": [[0], [1]],
        "selected_index": 0,
        "legal_action_count": 2,
    }


@pytest.fixture()
def actor_ckpt(tmp_path: Path) -> Path:
    rows = [_opening_row() for _ in range(6)] + [_academy_row() for _ in range(2)]
    result = run_stage_a_bc(
        rows, out_dir=tmp_path / "a", config=StageAConfig(epochs=1, batch_size=4)
    )
    return Path(result.checkpoint_path)


@pytest.mark.unit
def test_runtime_config_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POKEBOT_SLOWKING_DISTILL_ENABLED", raising=False)
    monkeypatch.delenv("POKEBOT_SLOWKING_DISTILL_MODE", raising=False)
    cfg = load_runtime_config_from_env()
    assert cfg.enabled is False
    assert cfg.mode is SlowkingDistillMode.DISABLED
    assert cfg.runs is False
    assert cfg.selects_actions is False


@pytest.mark.unit
def test_runtime_actor_and_critical_search(actor_ckpt: Path) -> None:
    cfg = SlowkingDistillRuntimeConfig(
        enabled=True,
        mode=SlowkingDistillMode.ACTIVE,
        actor_checkpoint=str(actor_ckpt),
        use_belief_mcts=True,
    )
    rt = SlowkingDistillRuntime(config=cfg, actor_checkpoint=actor_ckpt)
    assert rt.ready
    opening = attach_heuristic_features(_opening_row())
    d1 = rt.decide_row(opening)
    assert d1.action in [[0], [1], [2], [3]]
    assert d1.critical is True
    # Mock backend when BeliefMCTS deps missing
    assert d1.search_used is True
    assert d1.search_backend == "mock_top_k"

    # Fail-closed when search raises
    def boom(_obs, _legal):
        raise RuntimeError("search exploded")

    rt2 = SlowkingDistillRuntime(
        config=cfg, actor_checkpoint=actor_ckpt, search_fn=boom
    )
    d2 = rt2.decide_row(attach_heuristic_features(_academy_row()))
    assert d2.source == "fail_closed_actor"
    assert d2.action  # non-empty legal


@pytest.mark.unit
def test_runtime_shadow_does_not_select(actor_ckpt: Path) -> None:
    cfg = SlowkingDistillRuntimeConfig(
        enabled=True,
        mode=SlowkingDistillMode.SHADOW,
        actor_checkpoint=str(actor_ckpt),
    )
    rt = SlowkingDistillRuntime(config=cfg, actor_checkpoint=actor_ckpt)
    legal = [[0], [1]]
    obs = _academy_row()["observation"]
    action = rt.select_action(obs, legal)
    assert action == [0]  # shadow returns first legal, not search choice
    bridge = SlowkingDistillAgentBridge(runtime=rt, config=cfg)
    shadow = bridge.shadow(obs, legal, committed_action=[1])
    assert shadow is not None
    assert bridge.last_diagnostics.trace.get("shadow") is True


@pytest.mark.unit
def test_self_play_synthetic(actor_ckpt: Path, tmp_path: Path) -> None:
    result = run_population_self_play(
        actor_checkpoint=actor_ckpt,
        output_dir=tmp_path / "sp",
        games_per_opponent=1,
        max_turns=4,
        seed=1,
    )
    assert len(result.games) == 4  # 4 default opponents × 1 game
    assert 0.0 <= result.win_rate <= 1.0
    assert Path(result.receipt_path).is_file()
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["promoted"] is False
    assert receipt["research_only"] is True


@pytest.mark.unit
def test_promotion_never_self_promotes() -> None:
    decision = evaluate_promotion(
        PromotionRequest(
            eval_gate_passed=True,
            paired_win_delta=0.9,
            action_agreement=0.99,
            stage_d_win_rate=0.8,
            actor_checkpoint="/tmp/actor.pt",
            allow_external_research_tag=True,
            research_tag="slowking-distill-research",
        )
    )
    assert decision.promoted is False
    assert decision.research_tag_allowed is True
    assert decision.authority["selector_authority"] is False
    assert decision.authority["serving_authority"] is False


@pytest.mark.unit
def test_belief_search_resolves_mock() -> None:
    fn, backend = resolve_stage_c_search_fn(BeliefSearchBundle())
    assert backend == "mock_top_k"
    out = fn({"current": {}, "select": {}}, [[0], [1]])
    assert "chosen_action" in out
    assert out["search_backend"] == "mock_top_k"
