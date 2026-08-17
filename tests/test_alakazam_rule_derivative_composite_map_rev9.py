from __future__ import annotations

import pytest

from poke_bot.alakazam_rule_derivative_composite_map_rev9 import (
    CompositeMapError,
    build_tensor_source_map,
)


class Tensor:
    def __init__(self, shape: tuple[int, ...], dtype: str = "float32") -> None:
        self.shape = shape
        self.dtype = dtype


def test_r195_first_and_r274_addition_are_classified() -> None:
    value = build_tensor_source_map(
        r195_state={"backbone.weight": Tensor((2, 2))},
        r274_state={
            "backbone.weight": Tensor((2, 2)),
            "own_deck.weight": Tensor((3,)),
        },
    )
    assert value["tensor_count"] == 2
    assert value["source_counts"] == {
        "r195_exact_name_shape_dtype": 1,
        "r274_architecture_addition_absent_from_r195": 1,
    }
    assert [row["source"] for row in value["tensors"]] == [
        "r195_exact_name_shape_dtype",
        "r274_architecture_addition_absent_from_r195",
    ]


def test_same_name_shape_drift_fails_closed() -> None:
    with pytest.raises(CompositeMapError, match="shape drift"):
        build_tensor_source_map(
            r195_state={"backbone.weight": Tensor((3, 2))},
            r274_state={"backbone.weight": Tensor((2, 2))},
        )


def test_same_name_dtype_drift_fails_closed() -> None:
    with pytest.raises(CompositeMapError, match="dtype drift"):
        build_tensor_source_map(
            r195_state={"backbone.weight": Tensor((2, 2), "float16")},
            r274_state={"backbone.weight": Tensor((2, 2), "float32")},
        )


def test_unused_r195_name_fails_closed() -> None:
    with pytest.raises(CompositeMapError, match="r195 contains names absent"):
        build_tensor_source_map(
            r195_state={"old.weight": Tensor((1,))},
            r274_state={"new.weight": Tensor((1,))},
        )
