from __future__ import annotations

import numpy as np

from scripts.train_public_matchup_tree import _expanded_predict_proba


class _SparseClassModel:
    classes_ = np.asarray([0, 2, 5], dtype=np.int64)

    def predict_proba(self, _matrix):
        return np.asarray(
            [
                [0.10, 0.70, 0.20],
                [0.80, 0.05, 0.15],
            ],
            dtype=np.float64,
        )


def test_predict_proba_is_expanded_to_canonical_class_indexes() -> None:
    probabilities = _expanded_predict_proba(
        _SparseClassModel(),
        matrix=object(),
        class_count=7,
    )

    assert probabilities.shape == (2, 7)
    np.testing.assert_allclose(probabilities[:, 0], [0.10, 0.80])
    np.testing.assert_allclose(probabilities[:, 2], [0.70, 0.05])
    np.testing.assert_allclose(probabilities[:, 5], [0.20, 0.15])
    np.testing.assert_allclose(probabilities[:, [1, 3, 4, 6]], 0.0)


def test_out_of_range_fitted_class_is_rejected() -> None:
    model = _SparseClassModel()
    model.classes_ = np.asarray([0, 7], dtype=np.int64)

    try:
        _expanded_predict_proba(model, matrix=object(), class_count=7)
    except ValueError as exc:
        assert "outside canonical width" in str(exc)
    else:
        raise AssertionError("out-of-range class index was accepted")
