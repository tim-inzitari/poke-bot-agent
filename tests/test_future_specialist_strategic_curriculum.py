from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_future_specialist_strategic_curriculum import materialize


def test_final_format_curriculum_can_bind_optional_combo_route(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "guide.yaml"
    contract.write_text("specialist_id: alakazam\n", encoding="utf-8")
    ready = tmp_path / "ready.json"
    ready.write_text(
        json.dumps(
            {
                "status": "validated",
                "specialist_id": "alakazam",
                "guide_rows": 123,
            }
        ),
        encoding="utf-8",
    )
    implementation = tmp_path / "train.py"
    implementation.write_text("# training implementation\n", encoding="utf-8")

    outputs = materialize(
        specialist_id="alakazam",
        guide_contract=contract,
        guide_ready_receipt=ready,
        output_root=tmp_path / "out",
        training_implementation=implementation,
        include_combo_state=True,
    )

    role_map = json.loads(outputs["head_role_map"].read_text(encoding="utf-8"))
    spec = json.loads(outputs["curriculum_spec"].read_text(encoding="utf-8"))
    receipt = json.loads(
        outputs["validation_receipt"].read_text(encoding="utf-8")
    )

    assert len(role_map["canonical_learned_decision_sources"]) == 19
    assert role_map["canonical_learned_decision_sources"][-2:] == [
        "setup_board_outcome",
        "combo_state",
    ]
    assert role_map["heads"]["combo_state"]["fusion_role"] == "fused_input"
    assert (
        role_map["heads"]["combo_state"]["action_influence"]
        == "bounded_option_conditioned_route"
    )
    assert "combo_state" in spec["curriculum_heads"]
    assert "combo_state-route" in receipt["validated_route_ids"]
    assert receipt["route_validation"]["combo_state"][
        "post_training_route_gradient_norm"
    ] > 0.0
    assert receipt["action_influence_ablations"]["combo_state"][
        "mean_absolute_action_logit_delta"
    ] > 0.0
