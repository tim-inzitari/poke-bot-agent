from __future__ import annotations

import json
from pathlib import Path

import torch

from poke_bot.matchup_adapters_v6 import (
    ADAPTER_CHECKPOINT_FORMAT,
    SLOT_CAPACITY,
    load_slot_registry,
    registry_digest,
    slot_map,
)
from scripts.activate_public_matchup_tree import activate


def test_v6_runtime_activation_uses_the_embedded_19_route_registry(
    tmp_path: Path,
) -> None:
    registry = load_slot_registry()
    targets = list(registry["active_expert_ids"])
    adapter_config = {
        "format": ADAPTER_CHECKPOINT_FORMAT,
        "slot_capacity": SLOT_CAPACITY,
        "slot_registry_digest": registry_digest(registry),
        "slot_registry": registry,
    }
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "extra": {
                "matchup_adapter_config": adapter_config,
                "dormant_matchup_adapter_bank": {
                    "schema": "poke_bot.zero_dormant_matchup_adapter/v1",
                    "zero_output": True,
                    "runtime_enabled": False,
                    "adapter_config": adapter_config,
                },
                "dormant_matchup_adapter_fit": {"route_decisions": {}},
            }
        },
        checkpoint,
    )
    width = len(targets) + 1
    unknown = [0.0] * width
    unknown[-1] = 1.0
    source = tmp_path / "candidate.json"
    source.write_text(
        json.dumps(
            {
                "schema": "poke_bot.public_matchup_decision_tree/v1",
                "runtime_enabled": False,
                "targets": targets,
                "prediction_contract": {
                    "route_output_width": len(targets),
                    "route_class_names": targets,
                    "unknown_is_separate_abstention": True,
                    "unknown_class_index": len(targets),
                    "adapter_count": len(targets),
                },
                "validation": {
                    "classes": {
                        target: {
                            "precision": 1.0,
                            "weighted_support": 10_000,
                        }
                        for target in targets
                    }
                },
                "runtime_calibration": {
                    "per_archetype": {
                        target: {
                            "available": True,
                            "precision": 1.0,
                            "min_leaf_confidence": 0.9,
                        }
                        for target in targets
                    }
                },
                "tree": {
                    "class_names": [*targets, "unknown"],
                    "children_left": [-1],
                    "children_right": [-1],
                    "feature_card_id": [-2],
                    "threshold": [-2.0],
                    "weighted_class_counts": [unknown],
                    "node_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "runtime.json"

    result = activate(
        source,
        checkpoint,
        output,
        min_precision=0.93,
        min_support=10_000,
        min_leaf_confidence=0.9,
        consecutive_required=2,
        allow_zero_materialized_adapters=True,
    )

    activated = json.loads(output.read_text(encoding="utf-8"))
    runtime = activated["runtime_contract"]
    assert len(result["accepted_archetype_ids"]) == 19
    assert result["accepted_archetype_ids"] == targets
    assert result["accepted_archetype_ids"][-1] == "teal-mask-ogerpon-ex"
    assert runtime["route_target_ids"] == targets
    assert runtime["route_physical_slots"] == [
        slot_map(registry)[target] for target in targets
    ]
    assert runtime["physical_slot_capacity"] == SLOT_CAPACITY
    assert runtime["slot_registry_digest"] == registry_digest(registry)
    assert runtime["zero_materialized_adapters_allowed"] is True
