"""Torch-free target contract shared by Elmo materialization and training."""

from __future__ import annotations

import math
from typing import Any

COMBO_STATE_KEY = "combo_state"
COMBO_STATE_TARGET_SCHEMA = "poke_bot.slowking_combo_state_targets/v1"
TOP_DECK_CLASSES = 5
SEEK_SOURCE_CLASSES = 7
COPIED_ATTACK_WIDTH = 6
VISIBLE_PIECE_WIDTH = 5
ENERGY_ROUTE_WIDTH = 5
BENCH_CONTINUITY_WIDTH = 4
VECTOR_WIDTH = (
    COPIED_ATTACK_WIDTH
    + VISIBLE_PIECE_WIDTH
    + ENERGY_ROUTE_WIDTH
    + BENCH_CONTINUITY_WIDTH
)
COMBO_STATE_OUTPUT_WIDTH = TOP_DECK_CLASSES + SEEK_SOURCE_CLASSES + VECTOR_WIDTH


def validate_combo_state_labels(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("combo-state target must be a mapping")
    if value.get("schema") != COMBO_STATE_TARGET_SCHEMA:
        raise ValueError("combo-state target schema mismatch")

    def class_target(name: str, width: int) -> tuple[int, bool]:
        raw = value.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"combo-state {name} must be a mapping")
        mask = raw.get("mask")
        if not isinstance(mask, bool):
            raise ValueError(f"combo-state {name}.mask must be boolean")
        target = raw.get("target")
        if not mask:
            if target is not None:
                raise ValueError(f"masked combo-state {name} must have null target")
            return 0, False
        if isinstance(target, bool) or not isinstance(target, int):
            raise ValueError(f"combo-state {name}.target must be an integer")
        if not 0 <= target < width:
            raise ValueError(f"combo-state {name}.target is outside range")
        return int(target), True

    top_target, top_mask = class_target("top_deck", TOP_DECK_CLASSES)
    seek_target, seek_mask = class_target("seek_source", SEEK_SOURCE_CLASSES)
    raw_vector = value.get("vector")
    if not isinstance(raw_vector, dict):
        raise ValueError("combo-state vector must be a mapping")
    targets = raw_vector.get("target")
    masks = raw_vector.get("mask")
    if not isinstance(targets, list) or len(targets) != VECTOR_WIDTH:
        raise ValueError("combo-state vector target width mismatch")
    if not isinstance(masks, list) or len(masks) != VECTOR_WIDTH:
        raise ValueError("combo-state vector mask width mismatch")
    clean_targets: list[float] = []
    clean_masks: list[bool] = []
    for index, (target, mask) in enumerate(zip(targets, masks)):
        if not isinstance(mask, bool):
            raise ValueError(f"combo-state vector mask {index} is not boolean")
        if not mask:
            if target is not None:
                raise ValueError(
                    f"masked combo-state vector target {index} must be null"
                )
            clean_targets.append(0.0)
            clean_masks.append(False)
            continue
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise ValueError(f"combo-state vector target {index} is not numeric")
        number = float(target)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(
                f"combo-state vector target {index} is outside [0, 1]"
            )
        clean_targets.append(number)
        clean_masks.append(True)
    return {
        "top_deck_target": top_target,
        "top_deck_mask": top_mask,
        "seek_source_target": seek_target,
        "seek_source_mask": seek_mask,
        "vector_target": clean_targets,
        "vector_mask": clean_masks,
    }
