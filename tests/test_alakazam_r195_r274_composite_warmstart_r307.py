from __future__ import annotations

import pytest

from poke_bot.alakazam_r195_r274_composite_warmstart_r307 import (
    CompositeWarmstartError,
    assemble_composite_warmstart,
)


class Tensor:
    def __init__(self, value: int, shape: tuple[int, ...]) -> None:
        self.value = value
        self.shape = shape

    def detach(self) -> "Tensor":
        return self

    def clone(self) -> "Tensor":
        return Tensor(self.value, self.shape)


def test_r195_wins_r274_fills_absent_and_new_is_explicit() -> None:
    target = {
        "backbone.weight": Tensor(0, (2, 2)),
        "matchup_adapter.weight": Tensor(0, (2,)),
        "public_rule_adapter.weight": Tensor(0, (3,)),
    }
    result = assemble_composite_warmstart(
        target_state=target,
        r195_state={"backbone.weight": Tensor(195, (2, 2))},
        r274_state={
            "backbone.weight": Tensor(274, (2, 2)),
            "matchup_adapter.weight": Tensor(274, (2,)),
        },
        new_parameter_prefixes=("public_rule_adapter",),
        r274_checkpoint_sha256="sha256:" + "2" * 64,
    )
    assert result.state_dict["backbone.weight"].value == 195
    assert result.state_dict["matchup_adapter.weight"].value == 274
    assert result.state_dict["public_rule_adapter.weight"].value == 0
    assert result.source_map == {
        "backbone.weight": "r195",
        "matchup_adapter.weight": "r274_architecture_addition",
        "public_rule_adapter.weight": "new_initialization",
    }
    assert result.receipt["all_target_tensors_classified_exactly_once"] is True


def test_r195_shape_drift_fails_even_when_r274_would_fit() -> None:
    with pytest.raises(CompositeWarmstartError, match="r195 tensor shape drift"):
        assemble_composite_warmstart(
            target_state={"backbone.weight": Tensor(0, (2, 2))},
            r195_state={"backbone.weight": Tensor(195, (3, 2))},
            r274_state={"backbone.weight": Tensor(274, (2, 2))},
            new_parameter_prefixes=(),
            r274_checkpoint_sha256="sha256:" + "2" * 64,
        )


def test_unclassified_new_tensor_fails() -> None:
    with pytest.raises(CompositeWarmstartError, match="unclassified target tensor"):
        assemble_composite_warmstart(
            target_state={"mystery.weight": Tensor(0, (1,))},
            r195_state={},
            r274_state={},
            new_parameter_prefixes=("public_rule_adapter",),
            r274_checkpoint_sha256="sha256:" + "2" * 64,
        )
