"""Core → specialist curriculum stages for pure RL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CurriculumStage(str, Enum):
    CORE = "core"  # deck-agnostic Stage A
    SPECIALIST = "specialist"  # hammer-pult after warm-start
    SPECIALIST_WIDEN = "specialist_widen"  # Stage C
    SEARCH_DISTILL = "search_distill"  # Stage D later


@dataclass(frozen=True)
class StageConfig:
    stage: CurriculumStage
    our_deck_mode: str  # "multi_archetype" | "hammer-pult"
    opponent_mode: str  # "public_plus_roster" | "selfplay_meta" | "widen"
    mcts_sims: int = 0
    sample_actions: bool = True
    bootstrap_mix: float = 0.0
    target_heldout_wr: float = 0.70
    min_heldout_games: int = 200


STAGE_DEFAULTS: dict[CurriculumStage, StageConfig] = {
    CurriculumStage.CORE: StageConfig(
        stage=CurriculumStage.CORE,
        our_deck_mode="multi_archetype",
        opponent_mode="public_plus_roster",
    ),
    CurriculumStage.SPECIALIST: StageConfig(
        stage=CurriculumStage.SPECIALIST,
        our_deck_mode="hammer-pult",
        opponent_mode="selfplay_meta",
    ),
    CurriculumStage.SPECIALIST_WIDEN: StageConfig(
        stage=CurriculumStage.SPECIALIST_WIDEN,
        our_deck_mode="hammer-pult",
        opponent_mode="widen",
    ),
    CurriculumStage.SEARCH_DISTILL: StageConfig(
        stage=CurriculumStage.SEARCH_DISTILL,
        our_deck_mode="hammer-pult",
        opponent_mode="widen",
        mcts_sims=0,  # distill offline; collect stays search-off
        sample_actions=False,
    ),
}


def stage_for_iteration(
    *,
    core_gate_passed: bool,
    specialist_gate_passed: bool = False,
    widen: bool = False,
) -> StageConfig:
    if not core_gate_passed:
        return STAGE_DEFAULTS[CurriculumStage.CORE]
    if widen or specialist_gate_passed:
        return STAGE_DEFAULTS[CurriculumStage.SPECIALIST_WIDEN]
    return STAGE_DEFAULTS[CurriculumStage.SPECIALIST]


def stage_to_dict(cfg: StageConfig) -> dict[str, Any]:
    return {
        "stage": cfg.stage.value,
        "our_deck_mode": cfg.our_deck_mode,
        "opponent_mode": cfg.opponent_mode,
        "mcts_sims": cfg.mcts_sims,
        "sample_actions": cfg.sample_actions,
        "bootstrap_mix": cfg.bootstrap_mix,
        "target_heldout_wr": cfg.target_heldout_wr,
        "min_heldout_games": cfg.min_heldout_games,
    }
